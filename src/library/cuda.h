#ifndef CPUDA_CUDA_H
#define CPUDA_CUDA_H

#include <cstdio>   // printf
#include <stdlib.h> // this allows to use exit function in code, also size_t in library
#include "cuda_result.h" // file with CUresult and it's values definition

// ------------ structures ------------

struct CUctx_st{};
struct CUmod_st{};
struct CUfunc_st{};
struct CUstream_st{};

// ------------ definitions ------------

// --- type definitions like in real cuda.h library ---

#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
typedef unsigned long long CUdeviceptr;
#else
typedef unsigned int CUdeviceptr;
#endif
typedef int CUdevice;                                     /**< CUDA device */
typedef struct CUctx_st *CUcontext;                       /**< CUDA context */
typedef struct CUmod_st *CUmodule;                        /**< CUDA module */
typedef struct CUfunc_st *CUfunction;                     /**< CUDA function */
typedef struct CUstream_st *CUstream;                     /**< CUDA stream */

// --- functions prototypes ---

// initialization
CUresult cuInit(unsigned int Flags);
CUresult cuDeviceGet(CUdevice *device, int ordinal);
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);

// module/function loading
CUresult cuModuleLoad(CUmodule *module, const char *fname);
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);

// memory allocation/deallocation
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemAllocHost(void **pp, size_t bytesize);
CUresult cuMemFreeHost(void *p);

// memory copying
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
/*
// other ways of memory copying
CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyDtoA(CUarray dstArray, size_t dstOffset, CUdeviceptr srcDevice, size_t ByteCount);
CUresult cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray, size_t srcOffset, size_t ByteCount);
CUresult cuMemcpyHtoA(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount);
CUresult cuMemcpyAtoH(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount);
CUresult cuMemcpyAtoA(CUarray dstArray, size_t dstOffset, CUarray srcArray, size_t srcOffset, size_t ByteCount);
CUresult cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset, const void *srcHost, size_t ByteCount, CUstream hStream);
CUresult cuMemcpyAtoHAsync(void *dstHost, CUarray srcArray, size_t srcOffset, size_t ByteCount, CUstream hStream);

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream);
CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
*/

// kernel launch
CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX,unsigned int gridDimY,unsigned int gridDimZ,
                        unsigned int blockDimX,unsigned int blockDimY,unsigned int blockDimZ,
                        unsigned int sharedMemBytes,
                        CUstream hStream,
                        void **kernelParams,
                        void **extra);

// synchronization
CUresult cuCtxSynchronize(void);

// context destruction
CUresult cuCtxDestroy(CUcontext ctx);

// ------------ temporary implementation ------------

// initialization
/**
 * \brief Initialize the CUDA driver API
 *
 * Flags parameter should be 0. This function should be called before any other functions from library.
 * If not then any function will return CUDA_ERROR_NOT_INITIALIZED.
 */
CUresult cuInit(unsigned int Flags){
    if (Flags != 0)
        return CUDA_ERROR_INVALID_VALUE;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice *device, int ordinal){
    // check initialised
    // check ordinal
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev){
    printf("creating context\n");
    return CUDA_SUCCESS;
}

// module/function loading
CUresult cuModuleLoad(CUmodule *module, const char *fname){
    printf("want to load module %s\n",fname);
    return CUDA_SUCCESS;
}
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name){
    printf("want to get function %s\n",name);
    return CUDA_SUCCESS;
}

// memory allocation/deallocation
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize){
	return CUDA_SUCCESS;
}
CUresult cuMemFree(CUdeviceptr dptr){
	return CUDA_SUCCESS;
}
CUresult cuMemAllocHost(void **pp, size_t bytesize){
	return CUDA_SUCCESS;
}
CUresult cuMemFreeHost(void *p){
	return CUDA_SUCCESS;
}

// memory copying
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount){
    printf("copy from HOST to DEVICE\n");
	return CUDA_SUCCESS;
}
CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount){
    printf("copy from DEVICE to HOST\n");
	return CUDA_SUCCESS;
}

// kernel launch
CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX,unsigned int gridDimY,unsigned int gridDimZ,
                        unsigned int blockDimX,unsigned int blockDimY,unsigned int blockDimZ,
                        unsigned int sharedMemBytes,
                        CUstream hStream,
                        void **kernelParams,
                        void **extra){
    printf("KERNEL is launched!\n");
	return CUDA_SUCCESS;
}

// synchronization
CUresult cuCtxSynchronize(void){
    printf("context is synchronized\n");
	return CUDA_SUCCESS;
}

// context destruction
CUresult cuCtxDestroy(CUcontext ctx){
    printf("destroying context\n");
	return CUDA_SUCCESS;
}

#endif /* CPUDA_CUDA_H */
