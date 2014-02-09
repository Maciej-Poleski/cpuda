#ifndef CONTEXT_IMPL
#define CONTEXT_IMPL

#include <stack>

#include "device_impl.h"
#include "cuda_result.h"
using namespace cpuda;

// -------- class implementation --------

struct CUctx_st{
    CUdevice dev;
    std::thread *task;

    CUctx_st(CUdevice device): dev(device){
        task = nullptr;
    }
    bool isTaskRunning(){
        return (task != nullptr) && task->joinable();
    }
    void waitForTask(){
        if(isTaskRunning()){
            task->join();
        }
    }
    void setTaskToDevice(){
        task = &getDevice(dev).deviceThread;
    }
};

typedef struct CUctx_st *CUcontext;

// ---------------- data ----------------

std::stack<CUcontext> contexts; // stack for contexts management

// ----------- helper methods -----------

bool isCurrentContext(const CUcontext ctx){
    return !contexts.empty() && (contexts.top() == ctx);
}

/**
 * Common test for valid context presence. Checks if API is initialized and
 * if any context is active.
 */
CUresult hasValidContext(){
    if(!isCudaInitialized()) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(contexts.empty()) // test whether any context is active
        return CUDA_ERROR_INVALID_CONTEXT;
    return CUDA_SUCCESS;
}

/**
 * Returns current context. Call only when library has valid context (initialized
 * and stack not empty).
 */
CUcontext getCurrentContext(){
    return contexts.top();
}

/**
 * Returns reference to object representing device on which given context was created.
 */
CUdevice_st& getContextDevice(CUcontext ctx){
    return getDevice(ctx->dev);
}


// --------- API implementation ---------

CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev){ // currently flags mean nothing
    if(!isCudaInitialized()) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(!isDeviceOrdinalInRange(dev))
        return CUDA_ERROR_INVALID_VALUE;
    *pctx = new CUctx_st(dev); // create new context
    cuCtxPushCurrent(*pctx); // supplant current context by the newly created context
    return CUDA_SUCCESS;
}

/**
 * From API:
 * Destroys the CUDA context specified by ctx. If ctx is current to the calling thread then
 * ctx will also be popped from the current thread's context stack.
 */
CUresult cuCtxDestroy(CUcontext ctx){ // context destruction
    if(!isCudaInitialized()) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(isCurrentContext(ctx))
        cuCtxPopCurrent(nullptr);
    delete ctx;
    return CUDA_SUCCESS;
}

CUresult cuCtxPushCurrent(CUcontext ctx){
    if(!isCudaInitialized()) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    contexts.push(ctx);
    return CUDA_SUCCESS;
}

CUresult cuCtxPopCurrent(CUcontext *pctx){
    CUresult res = hasValidContext();
    if(res == CUDA_SUCCESS){ // if there is active context, pop it
        if(pctx != nullptr) // if pctx is given store (old) context in it
            *pctx = contexts.top();
        contexts.pop(); // pop context, making another current
    }
    return res;
}

/**
 * FROM API:
 * Returns in *pctx the CUDA context bound to the calling CPU thread. If no context is bound
 * to the calling CPU thread then *pctx is set to NULL and CUDA_SUCCESS is returned.
 */
CUresult cuCtxGetCurrent(CUcontext *pctx){
    if(!isCudaInitialized()) // test whether API is initialized
        return CUDA_ERROR_NOT_INITIALIZED;
    if(contexts.empty())
        *pctx = nullptr;
    else
        *pctx = contexts.top();
    return CUDA_SUCCESS;
}

CUresult cuCtxGetDevice(CUdevice *device){
    CUresult res = hasValidContext();
    if(res == CUDA_SUCCESS){
        if(device == nullptr)
            return CUDA_ERROR_INVALID_VALUE;
        *device = contexts.top()->dev;
    }
    return res;
}

/**
 * Synchronization with context task - probably device kernel launch.
 */
CUresult cuCtxSynchronize(){
    CUresult res = hasValidContext();
    if(res == CUDA_SUCCESS)
        contexts.top()->waitForTask();
    return res;
}

#endif // CONTEXT_IMPL
