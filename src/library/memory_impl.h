#ifndef MEMORY_IMPL
#define MEMORY_IMPL

#include <cstdlib>  // malloc, free
#include <cstring>  // memcpy

#include "context_impl.h"
#include "cuda_result.h"

#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
typedef unsigned long long CUdeviceptr;
#else
typedef unsigned int CUdeviceptr;
#endif

// ----------- helper methods -----------

CUresult internalSimpleMemcpy(void *dst, const void *src, size_t ByteCount){
    CUresult res = hasValidContext();
    if(res == CUDA_SUCCESS){
        if(dst == nullptr || src == nullptr || ByteCount == 0)
            return CUDA_ERROR_INVALID_VALUE;
        memcpy(dst, src, ByteCount);
    }
    return res;
}

// --------- API implementation ---------

// --- memory allocation/deallocation ---

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize){
    CUresult res = hasValidContext();
    if(res == CUDA_SUCCESS){
        if(bytesize == 0)
            return CUDA_ERROR_INVALID_VALUE;
        *dptr = (CUdeviceptr) malloc(bytesize);
    }
    return res;
}

CUresult cuMemFree(CUdeviceptr dptr){
    CUresult res = hasValidContext();
    if(res == CUDA_SUCCESS){
        if((void *)dptr == nullptr)
            return CUDA_ERROR_INVALID_VALUE;
        free((void*) dptr);
    }
    return res;
}

CUresult cuMemAllocHost(void **pp, size_t bytesize){
    CUresult res = hasValidContext();
    if(res == CUDA_SUCCESS){
        if(bytesize == 0)
            return CUDA_ERROR_INVALID_VALUE;
        *pp = malloc(bytesize);
    }
    return res;
}

CUresult cuMemFreeHost(void *p){
    CUresult res = hasValidContext();
    if(res == CUDA_SUCCESS){
        if(p == nullptr)
            return CUDA_ERROR_INVALID_VALUE;
        free(p);
    }
    return res;
}

// ----------- memory copying -----------

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount){
    return internalSimpleMemcpy((void*) dst,(void*) src, ByteCount);
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount){
    return internalSimpleMemcpy((void*) dstDevice, srcHost, ByteCount);
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount){
    return internalSimpleMemcpy(dstHost, (void*) srcDevice, ByteCount);
}

#endif // MEMORY_IMPL
