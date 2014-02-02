#ifndef DEVICE_IMPL
#define DEVICE_IMPL

#include <thread>
#include <vector>
#include <algorithm>

#include "dim3.h"
#include "ptx_module.h"
#include "cuda_result.h"

#include "device_config.h" // get device configuration
using namespace cpuda;

// -------- class implementation --------

/**
 * This class represents CpUDA device, which imitates GPU.
 */
class CUdevice_st{ // additional structure for object representing device
    friend struct CUctx_st;

    // ---------------- data ----------------

    // thread executing device tasks, especially kernel launches
    std::thread deviceThread;

    // parameters of prepared kernel launch
    Module *module;
    Function *kernel;
    void **args;
    dim3 gridDim;
    dim3 blockDim;

    // -------------- methods ---------------

    bool checkGridSize(const dim3 &gridDim){
        return checkDimensionsInRange(gridDim, MAX_GRID_DIMENSIONS) && checkTotalSizeInLimit(gridDim, MAX_NUMBER_OF_BLOCKS);
    }

    bool checkBlockSize(const dim3 &blockDim){
        return checkDimensionsInRange(blockDim, MAX_BLOCK_DIMENSIONS) && checkTotalSizeInLimit(blockDim, MAX_THREADS_PER_BLOCK);
    }

    /**
     * This code is run by block runners. Every runner processes sequentially given range of blocks.
     * For each block it runs simultaneously all block threads.
     */
    void launchBlocks(std::vector<dim3> *blocks){
        int blockSize = totalSize(blockDim);
        std::vector<std::thread> threads(blockSize);

        for(auto &blockIdx : *blocks){ // process blocks sequentially
            module->initializeBlock(blockIdx);
            int t_id = 0;
            for(int tz = 0; tz < blockDim.z; tz++) for(int ty = 0; ty < blockDim.y; ty++) for(int tx = 0; tx < blockDim.x; tx++){
                threads[t_id++] = std::thread(&Function::run, kernel, args, blockIdx, doDim3(tx,ty,tz));
            }
            for (auto &t : threads) { // wait until all threads of this block will finish
                t.join();
            }
            module->releaseBlock(blockIdx);
        }
    }

    /**
     * This code is run by device thread. It simulates CUDA kernel execution.
     */
    void runKernel(){
        module->initializeModule(gridDim, blockDim, WARP_SIZE);

        dim_t gridSize = totalSize(gridDim);
        dim_t blockSize = totalSize(blockDim);

        int concurrent = NUMBER_OF_MULTIPROCESSORS;
        int perThread = gridSize/concurrent;
        int rem = gridSize%concurrent;

        // distribute block ids per block runner
        std::vector<std::vector<dim3>> blocksDistribution(concurrent);
        int num = 0;
        for(int bz = 0; bz < gridDim.z; bz++) for(int by = 0; by < gridDim.y; by++) for(int bx = 0; bx < gridDim.x; bx++){
            blocksDistribution[num].push_back(doDim3(bx,by,bz));
            if(blocksDistribution[num].size() == perThread + (rem > 0)){
                num++;
                rem--;
            }
        }
        // run block runners
        std::vector<std::thread> blockRunners;
        for(int i=0; i<concurrent; i++){
            blockRunners.push_back(std::thread(&CUdevice_st::launchBlocks, this, &blocksDistribution[i]));
        }
        for (auto &runner : blockRunners) { // wait until all block runners will finish
            runner.join();
        }

        module->cleanupModule();
    }

public:

    /**
     * Simple check if device (its thread) is running.
     */
    bool isDeviceRunning(){
        return deviceThread.joinable();
    }

    /**
     * Prepares device represented by this object to launch given kernel. Device checks if it is able
     * to execute this kernel. This method must be called before launching kernel via launchKernel().
     */
    CUresult prepareDeviceToLaunch(Module *m, Function *f, void **args, const dim3 &gridDim, const dim3 &blockDim){
        if(isDeviceRunning()) // currently we do not support concurrent kernels execution
            return CUDA_ERROR_LAUNCH_FAILED;
        if(!checkGridSize(gridDim) || !checkBlockSize(blockDim))
            return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
        this->module = m;
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

typedef int CUdevice;

// ---------------- data ----------------

CUdevice_st *devices; // table with devices

bool cudaInitialized = false; // flag if library was initialized
int numOfAvailableDevices; // number of available devices, set at initialization

// ----------- helper methods -----------

/**
 * Returns this library state - initialized or not.
 */
bool isCudaInitialized(){
    return cudaInitialized;
}

/**
 * Recognize how many devices should be available. Simply returns NUMBER_OF_DEVICES
 * from configuration. If this number should be recognized in another way - change
 * this method.
 */
int internalDeviceGetCount(){ // if number of devices should be recognized
    return NUMBER_OF_DEVICES;
}

/**
 * Tests if given ordinal corresponds to one of devices. Call only when initialized.
 */
bool isDeviceOrdinalInRange(int ordinal){
    return (0 <= ordinal) && (ordinal < numOfAvailableDevices);
}

/**
 * Returns reference to device specified by given number. Call only when initialized,
 * with correct device number. Otherwise there would be an error.
 */
CUdevice_st& getDevice(int ordinal){
    return devices[ordinal];
}

// --------- API implementation ---------

/**
 * From API reference:
 *
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
    cudaInitialized = true; // set flag to initialized
    return CUDA_SUCCESS;
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

#endif // DEVICE_IMPL
