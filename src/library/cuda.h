#ifndef CPUDA_CUDA_H
#define CPUDA_CUDA_H

#include <cstdio>   // printf
#include <cstring>   // memcpy
#include <stack>
#include <vector>
#include <thread>
#include <mutex>
#include <stdlib.h> // this allows to use exit function in code, also size_t in library
#include "cuda_result.h" // file with CUresult and it's values definition

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

// --- functions prototypes ---

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

typedef unsigned int dim_t;
struct dim3{
    dim_t x; dim_t y; dim_t z;
    dim3():                     x(0), y(0), z(0){}
    dim3(dim_t x, dim_t y, dim_t z):  x(x), y(y), z(z){}
    dim_t totalSize() const { // unsigned long long ?
        return x * y * z;
    }
};

// ------ configuration ------

const int NUMBER_OF_DEVICES = 1;
const dim3 MAX_GRID_DIMENSIONS(32,32,32);
const dim3 MAX_BLOCK_DIMENSIONS(1024,1024,64);
const dim_t MAX_THREADS_PER_BLOCK = 1024;

// ------ structures ------

std::mutex mtx;           // mutex for critical section

void testKernel(void **args, const dim3 &gridDim, const dim3 &blockDim, const dim3 &blockIdx, const dim3 &threadIdx){
    mtx.lock();
    printf("CpUDA: #(%d,%d,%d) in block #(%d,%d,%d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
    mtx.unlock();
}
struct CUdevice_st{ // additional structure for object representing device

    CUdevice_st(){
        printf("Device structure created!\n");
    }

    // simulate threads in block - run concurrently all threads
    void launchBlock(CUfunction f, void **args, const dim3 &gridDim, const dim3 &blockDim, const dim3 &blockIdx){
        std::vector<std::thread> threads;
        for(int tz = 0; tz < blockDim.z; tz++) for(int ty = 0; ty < blockDim.y; ty++) for(int tx = 0; tx < blockDim.x; tx++){
            threads.push_back(std::thread(testKernel, args, gridDim, blockDim, blockIdx, dim3(tx,ty,tz)));
        }
        for (auto &t : threads) { // wait until this block will finish
            t.join();
        }
        printf("\n");
    }

    bool checkDimensionsInRange(const dim3 &dim, const dim3 &range){
        return (dim.x < range.x) && (dim.y < range.y) && (dim.z < range.z);
    }

    bool checkGridSize(const dim3 &gridDim){
        return checkDimensionsInRange(gridDim, MAX_GRID_DIMENSIONS);
    }

    bool checkBlockSize(const dim3 &blockDim){
        return checkDimensionsInRange(blockDim, MAX_BLOCK_DIMENSIONS) && blockDim.totalSize() <= MAX_THREADS_PER_BLOCK;
    }

    CUresult launchKernel(CUfunction f, void **args, dim3 gridDim, dim3 blockDim){
        if(!checkGridSize(gridDim) || !checkBlockSize(gridDim))
            return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
        const dim_t gridSize = gridDim.totalSize();
        std::vector<dim3> blocks(gridSize);

        // initialize blocks id
        int curBlock = 0;
        for(int bz = 0; bz < gridDim.z; bz++) for(int by = 0; by < gridDim.y; by++) for(int bx = 0; bx < gridDim.x; bx++){
            blocks[curBlock] = dim3(bx,by,bz);
            curBlock++;
        }
        // simulate blocks in synchronous way
        for(int i=0; i<gridSize; i++){
            launchBlock(f, args, gridDim, blockDim, blocks[i]);
        }

        return CUDA_SUCCESS;
    }
};

struct CUctx_st{
    CUdevice dev;

    CUctx_st(CUdevice device): dev(device){
        printf("created context with device set to %d\n", dev);
    }
    ~CUctx_st(){
        printf("destroying context of device %d\n", dev);
    }
};
struct CUmod_st{};
struct CUfunc_st{};
struct CUstream_st{};

// ------ data ------

bool cudaInitialized = false;

CUdevice_st *devices; // table with devices
int numOfAvailableDevices; // number of available devices

std::stack<CUcontext> contexts; // stack for contexts management

// ------ functions ------

int internalDeviceGetCount(){
    return NUMBER_OF_DEVICES;
}

// initialization
/**
 * \brief Initialize the CUDA driver API
 *
 * Flags parameter should be 0. This function should be called before any other functions from library.
 * If not then any function will return CUDA_ERROR_NOT_INITIALIZED.
 */
CUresult cuInit(unsigned int Flags){
    if(cudaInitialized) // nothing to do, already initialized
        return CUDA_SUCCESS;
    if (Flags != 0)
        return CUDA_ERROR_INVALID_VALUE;

    numOfAvailableDevices = internalDeviceGetCount(); // get number of available devices
    devices = new CUdevice_st[numOfAvailableDevices]; // initialize structures representing them
    cudaInitialized = true;
    return CUDA_SUCCESS;
}

bool isDeviceOrdinalInRange(int ordinal){
    return (0 <= ordinal) && (ordinal < numOfAvailableDevices);
}

CUresult cuDeviceGet(CUdevice *device, int ordinal){
    if(!cudaInitialized) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(!isDeviceOrdinalInRange(ordinal))
        return CUDA_ERROR_INVALID_VALUE;
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int* count){
    if(!cudaInitialized) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    *count = numOfAvailableDevices;
    return CUDA_SUCCESS;
}

// context management

bool isCurrentContext(const CUcontext ctx){
    return !contexts.empty() && (contexts.top() == ctx);
}

CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev){ // currently flags mean nothing
    if(!cudaInitialized) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(!isDeviceOrdinalInRange(dev))
        return CUDA_ERROR_INVALID_VALUE;
    *pctx = new CUctx_st(dev); // create new context
    cuCtxPushCurrent(*pctx); // supplant current context by the newly created context
    return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(CUcontext ctx){ // context destruction
    if(!cudaInitialized) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    /* Destroys the CUDA context specified by \p ctx.
     * If \p ctx is current to the calling thread then \p ctx will also be
     * popped from the current thread's context stack
     */
    if(isCurrentContext(ctx))
        cuCtxPopCurrent(nullptr);
    delete ctx;
	return CUDA_SUCCESS;
}

CUresult cuCtxPushCurrent(CUcontext ctx){
    if(!cudaInitialized) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    contexts.push(ctx);
	return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent(CUcontext *pctx){
    if(!cudaInitialized) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(contexts.empty()){
        return CUDA_ERROR_INVALID_CONTEXT; // no context to pop
    }
    if(pctx != nullptr) // if pctx is given store (old) context in it
        *pctx = contexts.top();
    contexts.pop(); // pop context, making another current
	return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext *pctx){
    if(!cudaInitialized) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(contexts.empty())
        *pctx = nullptr;
    else
        *pctx = contexts.top();
	return CUDA_SUCCESS;
}

CUresult cuCtxGetDevice(CUdevice *device){
    if(!cudaInitialized) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(contexts.empty())
        return CUDA_ERROR_INVALID_CONTEXT;
    if(device == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    *device = contexts.top()->dev;
	return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(){ // synchronization
    if(!cudaInitialized) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(contexts.empty())
        return CUDA_ERROR_INVALID_CONTEXT; // no context to pop
    printf("wait until context task will be finished\n");
    printf("...TO DO...!\n");
    // TODO !!! !!! !!!
	return CUDA_SUCCESS;
}

// module/function loading
CUresult cuModuleLoad(CUmodule *module, const char *fname){
    printf("want to load module %s\n",fname);
    printf("...TO DO...!\n");
    module = nullptr;
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name){
    printf("want to get function %s\n",name);
    printf("...TO DO...!\n");
    hfunc = nullptr;
    return CUDA_SUCCESS;
}
//CUresult cuModuleUnload ( CUmodule hmod );

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

// kernel launch
CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX,unsigned int gridDimY,unsigned int gridDimZ,
                        unsigned int blockDimX,unsigned int blockDimY,unsigned int blockDimZ,
                        unsigned int sharedMemBytes,
                        CUstream hStream,
                        void **kernelParams,
                        void **extra){
    if(!cudaInitialized) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(contexts.empty()) // test if any context is active
        return CUDA_ERROR_INVALID_CONTEXT;
    if(kernelParams != nullptr && extra != nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    // check somewhere handle (f)
    printf("try to launch KERNEL on device...\n");
    return devices[contexts.top()->dev].launchKernel(f, kernelParams,
                                                     dim3(gridDimX,gridDimY,gridDimZ),
                                                     dim3(blockDimX,blockDimY,blockDimZ));
}

#endif /* CPUDA_CUDA_H */
