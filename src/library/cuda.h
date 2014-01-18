#ifndef CPUDA_CUDA_H
#define CPUDA_CUDA_H

#include <cstdio>    // printf
#include <cstring>   // memcpy
#include <algorithm> // min
#include <stack>
#include <vector>
#include <thread>
#include <mutex>
#include <stdlib.h>  // this allows to use exit function in code, also size_t in library

#include "cuda_result.h" // CUresult and its values definition
#include "module.h"      // Module, Function, printf_mutex

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

#include "dim3.h"

// ------ configuration ------

const unsigned int NUMBER_OF_DEVICES = 1;
const dim3 MAX_GRID_DIMENSIONS = doDim3(65535,65535,65535);
const dim3 MAX_BLOCK_DIMENSIONS = doDim3(1024,1024,64);
const unsigned int MAX_NUMBER_OF_BLOCKS = 65535;
const unsigned int MAX_THREADS_PER_BLOCK = 1024;
const unsigned int NUMBER_OF_MULTIPROCESSORS = 4; // minimal number of blocks to run simultaneously

const unsigned int MAX_THREADS = NUMBER_OF_MULTIPROCESSORS * MAX_THREADS_PER_BLOCK; // how many threads can run together

// ------ structures ------

struct CUmod_st{
    Module *mod;

    CUmod_st(const char *fname){
        mod = new Module(fname);
    }
    ~CUmod_st(){
        if(mod != nullptr)
            delete mod;
    }
};
struct CUfunc_st{
    Function *func;
    CUmodule cuMod;

    CUfunc_st(const CUmodule module, const char *name){
        func = new Function(*module->mod, name);
        cuMod = module;
    }
    ~CUfunc_st(){
        if(func != nullptr)
            delete func;
    }
};
struct CUstream_st{};

/**
 * This class represents device.
 */
class CUdevice_st{ // additional structure for object representing device
    friend struct CUctx_st;

    std::thread deviceThread;
    CUfunction kernel;
    void **args;
    dim3 gridDim;
    dim3 blockDim;

    bool checkGridSize(const dim3 &gridDim){
        return checkDimensionsInRange(gridDim, MAX_GRID_DIMENSIONS) && checkTotalSizeInLimit(gridDim, MAX_NUMBER_OF_BLOCKS);
    }

    bool checkBlockSize(const dim3 &blockDim){
        return checkDimensionsInRange(blockDim, MAX_BLOCK_DIMENSIONS) && checkTotalSizeInLimit(blockDim, MAX_THREADS_PER_BLOCK);
    }

    /**
     * This code is run by block runners. Every runner process sequentially given range of blocks.
     * For each block it runs simultaneously all block threads.
     */
    void launchBlocks(std::vector<dim3> *blocks){
        int blockSize = totalSize(blockDim);
        std::vector<std::thread> threads(blockSize);

        for(auto &blockIdx : *blocks){
            kernel->cuMod->mod->initializeBlock(blockIdx);
            int t_id = 0;
            for(int tz = 0; tz < blockDim.z; tz++) for(int ty = 0; ty < blockDim.y; ty++) for(int tx = 0; tx < blockDim.x; tx++){
                threads[t_id++] = std::thread(&Function::run, kernel->func, args, blockIdx, doDim3(tx,ty,tz));
            }
            for (auto &t : threads) { // wait until this block will finish
                t.join();
            }
            kernel->cuMod->mod->releaseBlock(blockIdx);
        }
    }

    /**
     * This code is run by device thread. It simulates CUDA kernel execution.
     */
    void runKernel(){
        printf(">>>>> RUN KERNEL!\n");
        kernel->cuMod->mod->initializeModule(gridDim, blockDim);

        dim_t gridSize = totalSize(gridDim);
        dim_t blockSize = totalSize(blockDim);

        int concurrent = std::min(gridSize, MAX_THREADS/blockSize);
        int perThread = gridSize/concurrent;
        int rem = gridSize%concurrent;

        std::vector<std::vector<dim3>> blocksDistribution(concurrent);
        blocksDistribution.reserve(gridSize);

        // distribute ids per block runner
        int num = 0;
        for(int bz = 0; bz < gridDim.z; bz++) for(int by = 0; by < gridDim.y; by++) for(int bx = 0; bx < gridDim.x; bx++){
            blocksDistribution[num].push_back(doDim3(bx,by,bz));
            if(blocksDistribution[num].size() == perThread + (rem > 0)){
                num++;
                rem--;
            }
        }
        std::vector<std::thread> blockRunners;
        for(int i=0; i<concurrent; i++){
            blockRunners.push_back(std::thread(&CUdevice_st::launchBlocks, this, &blocksDistribution[i]));
        }
        for (auto &runner : blockRunners) { // wait until all block runners will finish
            runner.join();
        }
        kernel->cuMod->mod->cleanupModule();
        printf("<<<<< END KERNEL!\n");
    }
public:
    /**
     * Simple check if device (its thread) is running.
     */
    bool isDeviceRunning(){
        return deviceThread.joinable();
    }

    /**
     * Prepares device represented by this object to launch given kernel. Device check if it is able
     * to execute this kernel. This method must be called before launching kernel via launchKernel().
     */
    CUresult prepareDeviceToLaunch(CUfunction f, void **args, const dim3 &gridDim, const dim3 &blockDim){
        if(isDeviceRunning()) // currently we do not support concurrent kernels execution
            return CUDA_ERROR_LAUNCH_FAILED;
        if(!checkGridSize(gridDim) || !checkBlockSize(blockDim))
            return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
        this->kernel = f;
        this->args = args;
        this->gridDim = gridDim;
        this->blockDim = blockDim;
        return CUDA_SUCCESS;
    }

    /**
     * Simple method to launch kernel in device (asynchronously). Calling thread only starts device thread
     * and returns. Device thread starts launching all kernel blocks and threads and waits for all of them.
     */
    CUresult launchKernel(){
        deviceThread = std::thread(&CUdevice_st::runKernel, this);
    }
};

struct CUctx_st{
    CUdevice dev;
    std::thread *task;

    CUctx_st(CUdevice device): dev(device){
        printf("created context with device set to %d\n", dev);
        task == nullptr;
    }
    bool isTaskRunning(){
        return (task != nullptr) && task->joinable();
    }
    void waitForTask(){
        if(isTaskRunning())
            task->join();
    }
    void setTaskToDevice(CUdevice_st &device){
        task = &device.deviceThread;
    }
    ~CUctx_st(){
        printf("destroying context of device %d\n", dev);
    }
};

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
    contexts.top()->waitForTask();
    return CUDA_SUCCESS;
}

// module/function loading

/*
 * From API:
 * Loads the corresponding module into the current context.
 * The CUDA driver API does not attempt to lazily allocate the resources needed by a module; it fails
 * if the memory for functions and data (constant and global) needed by the module cannot be allocated.
 */
/**
 * Currently we only loads module - its data will be allocated before kernel call.
 */
CUresult cuModuleLoad(CUmodule *module, const char *fname){
    printf("want to load module %s\n",fname);
    if(!cudaInitialized) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(contexts.empty()) // test if any context is active
        return CUDA_ERROR_INVALID_CONTEXT;
    if(module == nullptr || fname == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    CUcontext ctx = contexts.top(); // get current context
    try {
        *module = new CUmod_st(fname);
        //ctx.addModule(*module); // load module into the context
    } catch (const std::runtime_error& re) {
        return CUDA_ERROR_NOT_FOUND;
    }
    return CUDA_SUCCESS;
}

//    CUfunc_st *CUfunction;
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name){
    printf("want to get function %s\n",name);
    if(!cudaInitialized) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(contexts.empty()) // test if any context is active
        return CUDA_ERROR_INVALID_CONTEXT;
    if(hfunc == nullptr || name == nullptr)
        return CUDA_ERROR_INVALID_VALUE;
    try {
        *hfunc = new CUfunc_st(hmod, name);
    } catch (const std::runtime_error& re) {
        return CUDA_ERROR_NOT_FOUND;
    }
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
    CUcontext ctx = contexts.top();
    CUdevice_st &dev = devices[ctx->dev];

    CUresult res = dev.prepareDeviceToLaunch(f, kernelParams,
                                             doDim3(gridDimX,gridDimY,gridDimZ),
                                             doDim3(blockDimX,blockDimY,blockDimZ));
    if(res == CUDA_SUCCESS){
        printf("launching KERNEL in device!\n");
        dev.launchKernel();
        ctx->setTaskToDevice(dev);
    }
    return res;
}

#endif /* CPUDA_CUDA_H */
