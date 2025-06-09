#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>

const int INF = ((1 << 30) - 1);
const int V = 7001;
static int Dist[V][V];
int n, m;
int ncpus;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    
    #pragma omp parallel for num_threads(ncpus) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];

    }
    fclose(file);
}
void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int main(int argc, char* argv[]) {
    input(argv[1]);
    ncpus = omp_get_max_threads();
    // cpu_set_t cpu_set;
    // sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    // ncpus = CPU_COUNT(&cpu_set);
    // printf("%d cpus available\n", ncpus);

    for (int k = 0; k < n; ++k) {
        #pragma omp parallel for num_threads(ncpus) schedule(static)
        for (int i = 0; i < n; ++i) {
            if (Dist[i][k] < INF) {
                for (int j = 0; j < n; ++j) {
                    if (Dist[k][j] < INF) {
                        int new_dist = Dist[i][k] + Dist[k][j];
                        if (new_dist < Dist[i][j]) {
                            Dist[i][j] = new_dist;
                        }
                    }
                }
            }
        }
    }
    output(argv[2]);
    return 0;
}
