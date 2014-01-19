#ifndef EXEC_IMPL
#define EXEC_IMPL

#include "module_impl.h"
#include "stream_impl.h"
#include "context_impl.h"
#include "device_impl.h"
#include "cuda_result.h"

// --------- API implementation ---------

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX,unsigned int gridDimY,unsigned int gridDimZ,
                        unsigned int blockDimX,unsigned int blockDimY,unsigned int blockDimZ,
                        unsigned int sharedMemBytes,
                        CUstream hStream,
                        void **kernelParams,
                        void **extra){
    CUresult res = hasValidContext();
    if(res == CUDA_SUCCESS){
        if(kernelParams != nullptr && extra != nullptr)
            return CUDA_ERROR_INVALID_VALUE;

        // TODO check somewhere handle (f)
        CUcontext ctx = getCurrentContext();
        CUdevice_st &dev = getContextDevice(ctx);

        res = dev.prepareDeviceToLaunch(f->cuMod->mod, f->func, kernelParams,
                                        doDim3(gridDimX,gridDimY,gridDimZ),
                                        doDim3(blockDimX,blockDimY,blockDimZ));
        if(res == CUDA_SUCCESS){
            dev.launchKernel();
            ctx->setTaskToDevice();
        }
    }
    return res;
}

#endif // EXEC_IMPL
