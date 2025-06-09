#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <algorithm>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <execution>
#include <algorithm>
// #include <nvtx3/nvToolsExt.h>
using namespace std;

void exchangeData(float *localData, int partLen, int rank, float *receiveData, int pairPartLen, int pairRank, float *updateData){
    if(rank < pairRank){// find small data
        int totalCompare = 0, pair_ind = 0, local_ind = 0;
        while(totalCompare < partLen){
            if(local_ind < partLen && (pair_ind >= pairPartLen || localData[local_ind]<receiveData[pair_ind])){
                updateData[totalCompare] = localData[local_ind];
                local_ind++;
            }
            else{
                updateData[totalCompare] = receiveData[pair_ind];
                pair_ind++;
            }
            totalCompare++;
        }
    }
    else{// find large data
        int pair_ind = pairPartLen-1, local_ind = partLen-1, totalCompare = partLen-1;
        while(totalCompare >= 0){
            if(local_ind >= 0 && (pair_ind<0 || localData[local_ind]>receiveData[pair_ind])){
                updateData[totalCompare] = localData[local_ind];
                local_ind--;
            }
            else{
                updateData[totalCompare] = receiveData[pair_ind];
                pair_ind--;
            }
            totalCompare--;
        }
    }
    // memcpy(localData, updateData, partLen * sizeof(float));
}


// void exchangeData(float *localData, int partLen, int rank, float *receiveData, int pairPartLen, int pairRank, float *updateData){
//     if(rank < pairRank){// find small data
//         int totalCompare = 0, pair_ind = 0, local_ind = 0;
//         while(totalCompare < partLen){
//             if(local_ind < partLen){
//                 if(pair_ind >= pairPartLen){
//                     while(totalCompare < partLen){
//                         updateData[totalCompare] = localData[local_ind];
//                         local_ind++;
//                         totalCompare++;
//                     }
//                     break;
//                 }
//                 else if(localData[local_ind] < receiveData[pair_ind]){
//                     updateData[totalCompare] = localData[local_ind];
//                     local_ind++;
//                 }
//                 else{
//                     updateData[totalCompare] = receiveData[pair_ind];
//                     pair_ind++;
//                 }
//             }
//             else{
//                 while(totalCompare < partLen){
//                     updateData[totalCompare] = receiveData[pair_ind];
//                     pair_ind++;
//                     totalCompare++;
//                 }
//                 break;
//             }
//             totalCompare++;
//         }
//     }
//     else{// find large data
//         int pair_ind = pairPartLen-1, local_ind = partLen-1, totalCompare = partLen-1;
//         while(totalCompare >= 0){
//             if(local_ind >= 0 && (pair_ind<0 || localData[local_ind]>receiveData[pair_ind])){
//                 updateData[totalCompare] = localData[local_ind];
//                 local_ind--;
//             }
//             else{
//                 updateData[totalCompare] = receiveData[pair_ind];
//                 pair_ind--;
//             }
//             totalCompare--;
//         }
//     }
// }

int main(int argc, char **argv)
{
    //nvtxRangePush("Main");
    long long n = atoll(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int part = n/size, remain = n%size, start = part*rank, partLen=part;
    if(remain!=0){
		if(rank < remain){
			start+=rank;
			partLen++;
		}
        else start+=remain;
	}

    float *localData = new float[partLen];
    float *receiveData = new float[partLen+1];
    float *updateData = new float[partLen];

    ////////// read file //////////
    MPI_File input_file, output_file;
    //nvtxRangePush("IO");
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float) * start, localData, partLen, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);
    //nvtxRangePop();

    ////////// sort local data //////////
    // std::sort(localData, localData + partLen);
    // std::sort(std::execution::par, localData, localData + partLen);
    // tbb::parallel_sort(localData, localData + partLen);

    //nvtxRangePush("sort");
    boost::sort::spreadsort::spreadsort(localData, localData + partLen);
    //nvtxRangePop();

    ////////// record odd-even phase rank pair //////////

    int odd_phase_pair_rank, even_phase_pair_rank, even_phase_pair_part_len, odd_phase_pair_part_len;
    odd_phase_pair_part_len = even_phase_pair_part_len = part;

    // odd_phase_pair_rank = (rank % 2 == 0) ? rank - 1 : rank + 1;
    // even_phase_pair_rank = (rank % 2 == 0) ? rank + 1 : rank - 1;
    // if (odd_phase_pair_rank >= 0 && odd_phase_pair_rank < size) {
    //     if (odd_phase_pair_rank < remain) {
    //         odd_phase_pair_part_len++;
    //     }
    // }
    // else {
    //     odd_phase_pair_rank = MPI_PROC_NULL;
    // }
    // if (even_phase_pair_rank >= 0 && even_phase_pair_rank < size) {
    //     if (even_phase_pair_rank < remain) {
    //         even_phase_pair_part_len++;
    //     }
    // }
    // else {
    //     even_phase_pair_rank = MPI_PROC_NULL;
    // }

    if(rank%2 == 0){
        odd_phase_pair_rank = rank-1;
        even_phase_pair_rank = rank+1;
    }
    else{
        odd_phase_pair_rank = rank+1;
        even_phase_pair_rank = rank-1;
    }

    if(odd_phase_pair_rank < 0 || odd_phase_pair_rank >= size){
        odd_phase_pair_rank = MPI_PROC_NULL;
        // odd_phase_pair_part_len = 0;
    }
    else if(odd_phase_pair_rank < remain){
        odd_phase_pair_part_len++;
    }

    if(even_phase_pair_rank < 0 || even_phase_pair_rank >= size){
        even_phase_pair_rank = MPI_PROC_NULL;
        // even_phase_pair_part_len = 0;
    }
    else if(even_phase_pair_rank < remain){
        even_phase_pair_part_len++;
    }
    // printf("rank: %d, even phase pair: %d, odd phase pair: %d\n", rank, even_phase_pair_rank, odd_phase_pair_rank);

    bool needExchange = true;
    float tmpData = 0;

    //////////// odd even sort //////////
    for(int i=0; i<=size; i++){//如果剛好最大的兩個數字都在最前面兩個index，最差情況從最左換到最右需要siz+1次
        needExchange = true;
        if(i % 2 == 0){//even phase
            if(even_phase_pair_rank != MPI_PROC_NULL){
                if(rank % 2 == 0){
                    // MPI_Sendrecv(localData+partLen-1, 1, MPI_FLOAT, even_phase_pair_rank, 0, receiveData, 1, MPI_FLOAT, even_phase_pair_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    MPI_Sendrecv(&localData[partLen-1], 1, MPI_FLOAT, even_phase_pair_rank, 0, &tmpData, 1, MPI_FLOAT, even_phase_pair_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    if(localData[partLen-1] <= tmpData)  needExchange = false;
                }
                else{
                    // MPI_Sendrecv(localData, 1, MPI_FLOAT, even_phase_pair_rank, 0, receiveData+even_phase_pair_part_len-1, 1, MPI_FLOAT, even_phase_pair_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    MPI_Sendrecv(&localData[0], 1, MPI_FLOAT, even_phase_pair_rank, 0, &tmpData, 1, MPI_FLOAT, even_phase_pair_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    if(localData[0] >= tmpData)  needExchange = false;

                }
                if(needExchange){
                    //nvtxRangePush("Communication");
                    MPI_Sendrecv(localData, partLen, MPI_FLOAT, even_phase_pair_rank, 0, receiveData, even_phase_pair_part_len, MPI_FLOAT, even_phase_pair_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //nvtxRangePop();

                    //nvtxRangePush("exchangeData");
                    exchangeData(localData, partLen, rank, receiveData, even_phase_pair_part_len, even_phase_pair_rank, updateData);
                    //nvtxRangePop();

                    //nvtxRangePush("Swap");
                    std::swap(localData, updateData);
                    //nvtxRangePop();
                // }
                }
            }
        }
        else{//odd phase
            if(odd_phase_pair_rank != MPI_PROC_NULL){
                if(rank %2 != 0){
                    // MPI_Sendrecv(localData+partLen-1, 1, MPI_FLOAT, odd_phase_pair_rank, 0, receiveData, 1, MPI_FLOAT, odd_phase_pair_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    MPI_Sendrecv(&localData[partLen-1], 1, MPI_FLOAT, odd_phase_pair_rank, 0, &tmpData, 1, MPI_FLOAT, odd_phase_pair_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    if(localData[partLen-1] <= tmpData)  needExchange = false;

                }
                else{
                    // MPI_Sendrecv(localData, 1, MPI_FLOAT, odd_phase_pair_rank, 0, receiveData+odd_phase_pair_part_len-1, 1, MPI_FLOAT, odd_phase_pair_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    MPI_Sendrecv(&localData[0], 1, MPI_FLOAT, odd_phase_pair_rank, 0, &tmpData, 1, MPI_FLOAT, odd_phase_pair_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    if(localData[0] >= tmpData)  needExchange = false;

                }
                if(needExchange){
                    //nvtxRangePush("Communication");
                    MPI_Sendrecv(localData, partLen, MPI_FLOAT, odd_phase_pair_rank, 0, receiveData, odd_phase_pair_part_len, MPI_FLOAT, odd_phase_pair_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    //nvtxRangePop();

                    //nvtxRangePush("exchangeData");
                    exchangeData(localData, partLen, rank, receiveData, odd_phase_pair_part_len, odd_phase_pair_rank, updateData);
                    //nvtxRangePop();

                    //nvtxRangePush("Swap");
                    std::swap(localData, updateData);
                    //nvtxRangePop();
                // }
                }
            }
        }
    }

    ////////// write file //////////
    //nvtxRangePush("IO");
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float) * start, localData, partLen, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);
    //nvtxRangePop();

    MPI_Finalize();
    delete[] localData;
    delete[] receiveData;
    delete[] updateData;
    //nvtxRangePop();
    return 0;
}