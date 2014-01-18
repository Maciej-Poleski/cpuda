#ifndef MODULE_IMPL
#define MODULE_IMPL

#include "ptx_module.h"

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



extern bool cudaInitialized;
extern std::stack<CUcontext> contexts; // stack for contexts management

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

#endif // MODULE_IMPL
