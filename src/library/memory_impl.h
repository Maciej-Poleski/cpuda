#ifndef MEMORY_IMPL
#define MEMORY_IMPL

#include <thread>
#include <vector>
#include <algorithm>

#include "dim3.h"

// memory allocation/deallocation
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize){
    if(!cudaInitialized) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(contexts.empty()) // test if any context is active
        return CUDA_ERROR_INVALID_CONTEXT;
    if(bytesize == 0)
        return CUDA_ERROR_INVALID_VALUE;
    *dptr = (CUdeviceptr) malloc(bytesize);
    printf( "Memory allocated under: %u\n",*dptr);
    return CUDA_SUCCESS;
}

CUresult cuMemFree(CUdeviceptr dptr){
    if(!cudaInitialized) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(contexts.empty()) // test if any context is active
        return CUDA_ERROR_INVALID_CONTEXT;
    if((void *)dptr == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    printf("Free memory under value: %u\n",dptr);
    free((void*) dptr);
    return CUDA_SUCCESS;
}

CUresult cuMemAllocHost(void **pp, size_t bytesize){
    if(!cudaInitialized) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(contexts.empty())
        return CUDA_ERROR_INVALID_CONTEXT;
    if(bytesize == 0)
        return CUDA_ERROR_INVALID_VALUE;
    *pp = malloc(bytesize);
    printf( "Host memory allocated under: %u\n",*pp);
    return CUDA_SUCCESS;
}

CUresult cuMemFreeHost(void *p){
    if(!cudaInitialized) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(contexts.empty()) // test if any context is active
        return CUDA_ERROR_INVALID_CONTEXT;
    if(p == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    printf("Free memory under value: %u\n",p);
    free(p);
    return CUDA_SUCCESS;
}

// memory copying
CUresult internalSimpleMemcpy(void *dst, const void *src, size_t ByteCount){
    if(!cudaInitialized) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(contexts.empty()) // test if any context is active
        return CUDA_ERROR_INVALID_CONTEXT;
    if(dst == nullptr || src == nullptr || ByteCount == 0)
        return CUDA_ERROR_INVALID_VALUE;
    memcpy(dst, src, ByteCount);
    return CUDA_SUCCESS;
}

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount){
    printf("cu memory copy\n");
    return internalSimpleMemcpy((void*) dst,(void*) src, ByteCount);
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount){
    printf("copy from HOST to DEVICE\n");
    return internalSimpleMemcpy((void*) dstDevice, srcHost, ByteCount);
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount){
    printf("copy from DEVICE to HOST\n");
    internalSimpleMemcpy(dstHost, (void*) srcDevice, ByteCount);
    return CUDA_SUCCESS;
}

#endif // MEMORY_IMPL
