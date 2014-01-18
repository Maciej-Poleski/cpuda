#ifndef DEVICE_IMPL
#define DEVICE_IMPL

#include <thread>
#include <vector>
#include <algorithm>

#include "dim3.h"
#include "ptx_module.h"

// ------ configuration ------

const unsigned int NUMBER_OF_DEVICES = 1;
const dim3 MAX_GRID_DIMENSIONS = doDim3(65535,65535,65535);
const dim3 MAX_BLOCK_DIMENSIONS = doDim3(1024,1024,64);
const unsigned int MAX_NUMBER_OF_BLOCKS = 65535;
const unsigned int MAX_THREADS_PER_BLOCK = 1024;
const unsigned int NUMBER_OF_MULTIPROCESSORS = 4; // minimal number of blocks to run simultaneously

const unsigned int MAX_THREADS = NUMBER_OF_MULTIPROCESSORS * MAX_THREADS_PER_BLOCK; // how many threads can run together



/**
 * This class represents device.
 */
class CUdevice_st{ // additional structure for object representing device
    friend struct CUctx_st;

    std::thread deviceThread;
    Module *module;
    Function *kernel;
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
            module->initializeBlock(blockIdx);
            int t_id = 0;
            for(int tz = 0; tz < blockDim.z; tz++) for(int ty = 0; ty < blockDim.y; ty++) for(int tx = 0; tx < blockDim.x; tx++){
                threads[t_id++] = std::thread(&Function::run, kernel, args, blockIdx, doDim3(tx,ty,tz));
            }
            for (auto &t : threads) { // wait until this block will finish
                t.join();
            }
            module->releaseBlock(blockIdx);
        }
    }

    /**
     * This code is run by device thread. It simulates CUDA kernel execution.
     */
    void runKernel(){
        printf(">>>>> RUN KERNEL!\n");
        module->initializeModule(gridDim, blockDim);

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
        module->cleanupModule();
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




extern CUdevice_st *devices; // table with devices
extern int numOfAvailableDevices; // number of available devices
extern bool cudaInitialized;


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

#endif // DEVICE_IMPL
