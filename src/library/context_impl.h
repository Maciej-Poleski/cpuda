#ifndef CONTEXT_IMPL
#define CONTEXT_IMPL

// context management

std::stack<CUcontext> contexts; // stack for contexts management
extern bool cudaInitialized;

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

#endif // CONTEXT_IMPL
