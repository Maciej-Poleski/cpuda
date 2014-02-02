#ifndef MODULE_IMPL
#define MODULE_IMPL

#include <stdexcept>

#include "ptx_module.h"
#include "context_impl.h"
#include "cuda_result.h"
using namespace cpuda;

// -------- class implementation --------

struct CUmod_st{
    Module *mod; // ptx module wrapper

    CUmod_st(const char *fname){
        mod = new Module(fname);
    }
    ~CUmod_st(){
        if(mod != nullptr)
            delete mod;
    }
};

struct CUfunc_st{
    Function *func; // ptx kernel function wrapper
    const CUmod_st *cuMod; // pointer to module containing this function wrapper

    CUfunc_st(const CUmod_st *module, const char *name){
        func = new Function(*module->mod, name);
        cuMod = module;
    }
    ~CUfunc_st(){
        if(func != nullptr)
            delete func;
    }
};

typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;

// ----------- helper methods -----------
// --------- API implementation ---------

/**
 * From API:
 * Loads the corresponding module into the current context.
 * The CUDA driver API does not attempt to lazily allocate the resources needed by a module; it fails
 * if the memory for functions and data (constant and global) needed by the module cannot be allocated.
 *
 * Currently we only loads module - its data will be allocated before kernel call.
 */
CUresult cuModuleLoad(CUmodule *module, const char *fname){
    CUresult res = hasValidContext();
    if(res == CUDA_SUCCESS){
        if(module == nullptr || fname == nullptr)
            return CUDA_ERROR_INVALID_VALUE;
        CUcontext ctx = getCurrentContext();
        try {
            *module = new CUmod_st(fname);
            //ctx.addModule(*module); // TODO!!! load module into the context
        } catch (const std::runtime_error& re) {
            return CUDA_ERROR_NOT_FOUND;
        }
    }
    return res;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name){
    CUresult res = hasValidContext();
    if(res == CUDA_SUCCESS){
        if(hfunc == nullptr || name == nullptr)
            return CUDA_ERROR_INVALID_VALUE;
        try {
            *hfunc = new CUfunc_st(hmod, name);
        } catch (const std::runtime_error& re) {
            return CUDA_ERROR_NOT_FOUND;
        }
    }
    return res;
}
//CUresult cuModuleUnload ( CUmodule hmod ); // TODO!!!

#endif // MODULE_IMPL
