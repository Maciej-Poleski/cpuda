#ifndef EXEC_IMPL
#define EXEC_IMPL

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

    CUresult res = dev.prepareDeviceToLaunch(f->cuMod->mod, f->func, kernelParams,
                                             doDim3(gridDimX,gridDimY,gridDimZ),
                                             doDim3(blockDimX,blockDimY,blockDimZ));
    if(res == CUDA_SUCCESS){
        printf("launching KERNEL in device!\n");
        dev.launchKernel();
        ctx->setTaskToDevice(dev);
    }
    return res;
}

#endif // EXEC_IMPL
