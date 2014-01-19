/*
 * File for cuda_usage (and compiler) tests.
 */

extern "C"
{

#include<cstdio>

__device__
void thread_print(int tID, int bID){ // thread id, block id
    char tabs[] = {'\t', '\t', '\t', '\t','\t', '\t', '\t', '\t'};
    tabs[bID%8] = 0;
    printf("%s%4d\n",tabs,tID);
}

__global__
void cuFun(int x){
    int tID = (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + threadIdx.x;
    if(tID & 127) // print every 128th
        return;
    int bID = (blockIdx.z * gridDim.y * gridDim.x) + (blockIdx.y * gridDim.x) + blockIdx.x;
    thread_print(tID, bID);
}


}
