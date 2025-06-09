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
#include <pthread.h>
#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <nvtx3/nvToolsExt.h>

int iters;
double left;
double right;
double lower;
double upper;
int width;
int height;
int next_row = 0;
int* image;
pthread_mutex_t task_mutex = PTHREAD_MUTEX_INITIALIZER;

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

void computeMandelbrot_SIMD(int row) {
    const int simd_width = 8;  // 512 bits / 64 bits
    int simd_end = width - (width % (2 * simd_width));
    __m512d vec_left = _mm512_set1_pd(left);
    __m512d vec_right_left_diff = _mm512_set1_pd((right - left) / width);
    __m512d vec_y0 = _mm512_set1_pd(row * ((upper - lower) / height) + lower);
    __m512d vec_length_squared_limit = _mm512_set1_pd(4.0);
    __m512d vec_two = _mm512_set1_pd(2.0);

    for (int i = 0; i < simd_end; i += 2 * simd_width) {
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

        double result1[simd_width];
        double result2[simd_width];
        _mm512_storeu_pd(result1, vec_repeats1);
        _mm512_storeu_pd(result2, vec_repeats2);
        for (int j = 0; j < simd_width; ++j) {
            image[row * width + i + j] = (int)result1[j];
            image[row * width + i + simd_width + j] = (int)result2[j];
        }
    }

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
        image[row * width + i] = repeats;
    }
}



/* mandelbrot set */
void* getRow(void* arg){

    while(1) {
        int row;

        pthread_mutex_lock(&task_mutex);
        if (next_row >= height) {
            pthread_mutex_unlock(&task_mutex);
            break;
        }
        row = next_row++;
        pthread_mutex_unlock(&task_mutex);

        computeMandelbrot_SIMD(row);
    }
  	pthread_exit(NULL);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    nvtxRangePush("Main");
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int ncpus = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", ncpus);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    pthread_t threads[ncpus];

    // pthread_spin_init(&spinlock, PTHREAD_PROCESS_PRIVATE);

    nvtxRangePush("Compute");
    for(int i=0; i<ncpus; i++){
        int rc = pthread_create(&threads[i], NULL, getRow, NULL);
		if(rc != 0){
			printf("pthread_create error\n");
			return 1;
		}
    }
    // pthread_spin_destroy(&spinlock);
    for (int i=0; i<ncpus; i++) {
		pthread_join(threads[i], NULL);
	}
    nvtxRangePop();


    /* draw and cleanup */
    nvtxRangePush("IO");
    write_png(filename, iters, width, height, image);
    nvtxRangePop();

    free(image);

    nvtxRangePop();
}
