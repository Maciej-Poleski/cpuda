#ifndef CPUDA_CUDA_H
#define CPUDA_CUDA_H

#include <cstdio>    // printf
#include <cstring>   // memcpy
#include <algorithm> // min
#include <stack>
#include <stdlib.h>  // this allows to use exit function in code, also size_t in library

#include "cuda_result.h" // CUresult and its values definition

// ------------ definitions ------------

// --- structures like in real cuda.h library ---

struct CUctx_st;
struct CUmod_st;
struct CUfunc_st;
struct CUstream_st;

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

// --- functions prototypes from real cuda.h library ---

// initialization
CUresult cuInit(unsigned int Flags);
CUresult cuDeviceGet(CUdevice *device, int ordinal);
CUresult cuDeviceGetCount(int* count);

// context management
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev);
CUresult cuCtxDestroy(CUcontext ctx);
CUresult cuCtxPushCurrent(CUcontext ctx);
CUresult cuCtxPopCurrent(CUcontext *pctx);
CUresult cuCtxGetCurrent(CUcontext *pctx);
CUresult cuCtxGetDevice(CUdevice *device);
CUresult cuCtxSynchronize(void);

// module/function loading
CUresult cuModuleLoad(CUmodule *module, const char *fname);
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name);

// memory allocation/deallocation
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemAllocHost(void **pp, size_t bytesize);
CUresult cuMemFreeHost(void *p);

// memory copying
CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount);
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
/*
// other ways of memory copying
CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount); // asynchronous
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


// ------------ temporary implementation ------------

#include "device_impl.h"
#include "context_impl.h"
#include "module_impl.h"      // Module, Function
#include "memory_impl.h"
#include "exec_impl.h"

// ------ data ------

bool cudaInitialized = false;

CUdevice_st *devices; // table with devices
int numOfAvailableDevices; // number of available devices


#endif /* CPUDA_CUDA_H */
